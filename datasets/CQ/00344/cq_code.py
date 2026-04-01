import cadquery as cq

points = [(-382.5, 0), (-286.875, 0), (-191.25, 0), (-95.625, 0), (0.0, 0), (95.625, 0), (191.25, 0), (286.875, 0), (382.5, 0), (0, 227.5), (0, -227.5)]
result = (
cq.Workplane()
.rect(770, 460)
.extrude(5)
.pushPoints(points)
.rect(5, 5)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')