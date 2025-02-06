import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import nearest_points


def demo_intersection(
    target_eef_xy,
    arm_reaching_max,
    arm_reaching_min,
    arm_reaching_prefer,
    cur_base_xy,
    primary_move_dir,
):
    # Create circles and ring
    center = Point(target_eef_xy)
    outer_circle = center.buffer(arm_reaching_max)
    inner_circle = center.buffer(arm_reaching_min)
    prefer_circle = center.buffer(arm_reaching_prefer)

    ring = outer_circle.difference(inner_circle)

    # Create the line extending in the primary_move_dir
    line_length = 3  # Arbitrary length to ensure it intersects with the rings
    line_end = cur_base_xy.coords[0] + line_length * primary_move_dir
    line = LineString([cur_base_xy.coords[0], line_end])

    # Find intersection points
    intersection_points = []
    # intersection = line.intersection(ring)
    intersection = line.intersection(prefer_circle)

    if not intersection.is_empty:
        if intersection.geom_type == "Point":
            intersection_points.append(intersection)
        elif intersection.geom_type == "MultiPoint":
            intersection_points.extend(intersection.geoms)
        elif intersection.geom_type == "LineString":
            intersection_points.extend(
                [Point(coords) for coords in intersection.coords]
            )
        elif isinstance(intersection, MultiLineString):
            for linestring in intersection.geoms:
                intersection_points.extend(
                    [Point(coords) for coords in linestring.coords]
                )

    vis_basic_intersection(
        intersection_points,
        line,
        outer_circle,
        inner_circle,
        prefer_circle,
        cur_base_xy,
    )
    plt.show()
    return (
        intersection,
        intersection_points,
        outer_circle,
        inner_circle,
        prefer_circle,
        ring,
    )


def vis_basic_intersection(
    intersection_points, line, outer_circle, inner_circle, prefer_circle, cur_base_xy
):
    # Visualize the results
    fig, ax = plt.subplots()

    # Plot the initial point
    ax.plot(*cur_base_xy.xy, "bo", label="Current Base XY")

    # Plot the direction line
    line_x, line_y = line.xy
    ax.plot(line_x, line_y, "g--", label="Primary Move Direction")

    # Plot the outer_circle
    x_outer, y_outer = outer_circle.exterior.xy
    ax.plot(x_outer, y_outer, "r-", label="Outer Circle")

    # Plot the inner_circle
    x_inner, y_inner = inner_circle.exterior.xy
    ax.plot(x_inner, y_inner, "b-", label="Inner Circle")

    # Plot the prefer_circle
    x_prefer, y_prefer = prefer_circle.exterior.xy
    ax.plot(x_prefer, y_prefer, "y-", label="Prefer Circle")

    # Plot intersection points
    if intersection_points:
        for point in intersection_points:
            ax.plot(*point.xy, "rx", label="Intersection Point")
    else:
        # No intersection points, project centers
        projected_points = []
        # plot the prefer_circle
        center = prefer_circle.centroid
        point_on_line = nearest_points(center, line)[1]
        projected_points.append(point_on_line)
        ax.plot(*center.xy, "mo", label="Ring Center")
        ax.plot(*point_on_line.xy, "cx", label="Projected Center")

    # Configure the plot
    ax.set_aspect("equal")
    ax.legend()
    ax.set_xlim(2, 5)
    ax.set_ylim(2, 5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Intersections and Projections with Rings")
    plt.grid(True)
    return fig, ax
